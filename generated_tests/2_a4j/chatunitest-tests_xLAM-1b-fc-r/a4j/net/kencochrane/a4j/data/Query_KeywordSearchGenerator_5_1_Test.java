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

public class Query_KeywordSearchGenerator_5_1_Test {

    @Test
    public void testKeywordSearchGenerator() {
        // given
        Query query = new Query();
        String searchTerm = "test";
        String productLine = "product";
        String type = "type";
        String page = "1";
        // when
        String result = query.KeywordSearchGenerator(searchTerm, productLine, type, page);
        // then
        assertEquals("http://test.com?t=test&dev-t=DSB0XDDW1GQ3S&KeywordSearch=test&mode=product&type=type&page=1&f=xml", result);
    }
}
