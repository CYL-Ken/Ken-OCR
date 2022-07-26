#include "CgmhData.hpp"

#include <stdlib.h>
#include <filesystem>

static size_t WriteCallback(char *contents, size_t size, size_t nmemb, void *userp)
{
	((std::string*)userp)->append((char*)contents, size * nmemb);
	return size * nmemb;
}

cgmhDataSender::cgmhDataSender() 
{
    
}

cgmhDataSender::~cgmhDataSender() 
{
    
}

int cgmhDataSender::getI500Data(string i500Ip, string fileDst)
{
    string scpDst = "root@" + i500Ip + ":" + fileDst;
    // string removeKey = "ssh-keygen -f \"/home/pi/.ssh/known_hosts\" -R \"" + "192.168.53.207" + "\"";
    // system("ssh-keygen -f \"/home/pi/.ssh/known_hosts\" -R \"192.168.53.207\"");
    string command = "scp -o \"StrictHostKeyChecking no\" " + scpDst + " .";
    cout << command << endl;
    system(command.c_str());
    // system("scp -o \"StrictHostKeyChecking no\" OCR.png root@192.168.50.229:/data");   
    system("echo \"get i500result done\"");
    string fileName = fileDst.substr(fileDst.find('/', 1) + 1 , -1);
    ifstream inFile;
    inFile.open(fileName); //open the input file
    stringstream strStream;
    strStream << inFile.rdbuf(); //read the file
    string str = strStream.str(); //str holds the content of the file
    str.erase(std::remove(str.begin(), str.end(), '\n'), str.end());
    string delimiter = ",";
    this->_maskOn = str.substr(0, str.find(delimiter));
    this->_temperature = str.substr(str.find(delimiter) + 1 , -1);
    // cout << "i500 Data: " << this->_maskOn << " ; " << this->_temperature << endl;
    inFile.close();
    return 0;
}

int cgmhDataSender::genData(string lastName, string timeStamp, string personId)
{
    Json::Value dataValue;
    dataValue["userid"] = "abc";
    dataValue["password"] = "123456";
    dataValue["machine_name"] = "wow";
    dataValue["personID"] = personId;
    dataValue["personName"] = lastName;
    dataValue["temperature"] = this->_temperature;
    dataValue["timeStamp"] = timeStamp;
    dataValue["maskOn"] = this->_maskOn;
    dataValue["travelHistory"] = "";
    Json::StreamWriterBuilder builder;
    builder["commentStyle"] = "None";
    builder["indentation"] = "   ";
    this->_nameDataJson = timeStamp + "_data.json";
    this->_nameFaceImg = timeStamp + "_face.jpg";
    this->_nameIdCardImg = timeStamp + "_idCard.jpg";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    std::ofstream outputFileStream(this->_nameDataJson);
    writer -> write(dataValue, &outputFileStream);
}



int cgmhDataSender::writeRequiredImg(cv::Mat faceFrame, cv::Mat idCardFrame) 
{
    vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);  //選擇jpeg
    compression_params.push_back(85); //在這個填入你要的圖片質量

    cv::imwrite(this->_nameFaceImg, faceFrame, compression_params);
    cv::imwrite(this->_nameIdCardImg, idCardFrame, compression_params);
}

int cgmhDataSender::sendCgmhData(string cgmhUrl) 
{
    CURL *curl;
    CURLcode res;
    string readBuffer;

    curl = curl_easy_init();
    if(curl)
    {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "POST");
        curl_easy_setopt(curl, CURLOPT_URL, cgmhUrl.c_str());
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_DEFAULT_PROTOCOL, "http");
        struct curl_slist *headers = NULL;
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_mime *mime;
        curl_mimepart *part;
        mime = curl_mime_init(curl);
        part = curl_mime_addpart(mime);
        curl_mime_name(part,  "json_data");
        curl_mime_filedata(part, this->_nameDataJson.c_str());
        part = curl_mime_addpart(mime);
        curl_mime_name(part, "NHImage1");
        curl_mime_filedata(part, this->_nameIdCardImg.c_str());
        part = curl_mime_addpart(mime);
        curl_mime_name(part, "NowImage2");
        curl_mime_filedata(part, this->_nameFaceImg.c_str());
        curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);

        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        res = curl_easy_perform(curl);
        curl_mime_free(mime);
    }
    curl_easy_cleanup(curl);
    Json::Reader reader;
    Json::Value output;
    reader.parse(readBuffer, output);
    string removeCommand = "rm " + this->_nameDataJson + " " + this->_nameFaceImg + " " + this->_nameIdCardImg;
    system(removeCommand.c_str());
    cout << output << endl;
}


int cgmhDataSender::sendCgmhDataTEST(string cgmhUrl,string _JsonFile) 
{
    CURL *curl;
    CURLcode res;
    string readBuffer;

    curl = curl_easy_init();
    if(curl)
    {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "POST");
        curl_easy_setopt(curl, CURLOPT_URL, cgmhUrl.c_str());
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_DEFAULT_PROTOCOL, "http");
        struct curl_slist *headers = NULL;
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_mime *mime;
        curl_mimepart *part;
        mime = curl_mime_init(curl);
        part = curl_mime_addpart(mime);
        curl_mime_name(part, "json_data");
        curl_mime_filedata(part, _JsonFile.c_str());
        part = curl_mime_addpart(mime);
        curl_mime_name(part, "NHImage1");
        curl_mime_filedata(part, this->_nameIdCardImg.c_str());
        part = curl_mime_addpart(mime);
        curl_mime_name(part, "NowImage2");
        curl_mime_filedata(part, this->_nameFaceImg.c_str());
        curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);

        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);

        res = curl_easy_perform(curl);
        curl_mime_free(mime);
    }
    curl_easy_cleanup(curl);
    Json::Reader reader;
    Json::Value output;
    reader.parse(readBuffer, output);
    string removeCommand = "rm " + _JsonFile + " " + this->_nameFaceImg + " " + this->_nameIdCardImg;
    system(removeCommand.c_str());
    // cout << output << endl;
}
