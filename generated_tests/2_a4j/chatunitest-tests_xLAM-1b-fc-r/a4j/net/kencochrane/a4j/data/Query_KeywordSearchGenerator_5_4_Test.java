package net.kencochrane.a4j.data;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
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
public class Query_KeywordSearchGenerator_5_4_Test {

    @InjectMocks
    Query query;

    @Test
    public void testKeywordSearchGenerator() {
        String searchTerm = "test";
        String productLine = "test";
        String type = "test";
        String page = "1";
        String expectedUrl = "http://example.com/?t=12345&dev-t=DSB0XDDW1GQ3S&KeywordSearch=test&mode=test&type=test&page=1&f=xml";
        assertEquals(expectedUrl, query.KeywordSearchGenerator(searchTerm, productLine, type, page));
    }
}
