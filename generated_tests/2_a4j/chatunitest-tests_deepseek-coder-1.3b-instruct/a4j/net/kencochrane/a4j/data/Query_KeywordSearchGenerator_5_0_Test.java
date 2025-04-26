package net.kencochrane.a4j.data;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.LoadProperties;
import net.kencochrane.a4j.util.a4jUtil;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.Properties;

public class Query_KeywordSearchGenerator_5_0_Test {

    @Test
    public void testKeywordSearchGenerator() {
        Query query = new Query();
        query.serverURL = "http://www.example.com";
        query.associatesID = "12345";
        query.token = "DSB0XDDW1GQ3S";
        query.searchType = "Search";
        query.type = "Product";
        query.page = "1";
        query.offer = "10";
        query.searchValues = new ArrayList();
        query.searchValues.add("Test");
        query.searchValues.add("Product");
        String result = query.KeywordSearchGenerator("Test", "Product", "Search", "1");
        assertEquals("http://www.example.com?t=12345&dev-t=DSB0XDDW1GQ3S&KeywordSearch=Test&mode=Product&type=Search&page=1&f=xml", result);
    }
}
