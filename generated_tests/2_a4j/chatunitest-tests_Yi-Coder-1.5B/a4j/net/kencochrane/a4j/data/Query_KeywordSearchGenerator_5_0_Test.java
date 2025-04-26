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

@ExtendWith(MockitoExtension.class)
public class Query_KeywordSearchGenerator_5_0_Test {

    // Test class
    @Test
    public void testKeywordSearchGenerator() {
        Query query = new Query();
        query.serverURL = "http://www.example.com/products";
        query.associatesID = "12345";
        query.searchType = "KeywordSearch";
        query.type = "Software";
        query.page = "1";
        // query.offer = "1";
        String result = query.KeywordSearchGenerator("Java", "Java", "Software", "1");
        assertEquals("http://www.example.com/products?t=12345&dev-t=DSB0XDDW1GQ3S&KeywordSearch=Java&mode=Java&type=Software&page=1&f=xml", result);
    }
}
