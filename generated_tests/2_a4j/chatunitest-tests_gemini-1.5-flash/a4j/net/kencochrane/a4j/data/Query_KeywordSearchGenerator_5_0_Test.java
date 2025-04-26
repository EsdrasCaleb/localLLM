package net.kencochrane.a4j.data;

import java.lang.reflect.Field;
import java.util.ArrayList;
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
import java.util.Properties;

@ExtendWith(MockitoExtension.class)
class Query_KeywordSearchGenerator_5_0_Test {

    @Mock
    a4jUtil mockUtil;

    @Test
    void testKeywordSearchGenerator() throws Exception {
        Query query = new Query();
        // Use reflection to set private fields for testing
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "https://example.com/search");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "12345");
        Field utilField = Query.class.getDeclaredField("jawsUtil");
        utilField.setAccessible(true);
        utilField.set(query, mockUtil);
        when(mockUtil.encodeString(anyString())).thenAnswer(invocation -> {
            Object[] args = invocation.getArguments();
            return args[0].toString();
        });
        // Test case 1: Normal case
        String url1 = query.KeywordSearchGenerator("test", "productA", "typeA", "1");
        assertEquals("https://example.com/search?t=12345&dev-t=DSB0XDDW1GQ3S&KeywordSearch=test&mode=productA&type=typeA&page=1&f=xml", url1);
        // Test case 2: Empty search term
        String url2 = query.KeywordSearchGenerator("", "productB", "typeB", "2");
        assertEquals("https://example.com/search?t=12345&dev-t=DSB0XDDW1GQ3S&KeywordSearch=&mode=productB&type=typeB&page=2&f=xml", url2);
        // Test case 3: Null search term
        String url3 = query.KeywordSearchGenerator(null, "productC", "typeC", "3");
        assertEquals("https://example.com/search?t=12345&dev-t=DSB0XDDW1GQ3S&KeywordSearch=&mode=productC&type=typeC&page=3&f=xml", url3);
        // Test case 4:  Different page number
        String url4 = query.KeywordSearchGenerator("test4", "productD", "typeD", "10");
        assertEquals("https://example.com/search?t=12345&dev-t=DSB0XDDW1GQ3S&KeywordSearch=test4&mode=productD&type=typeD&page=10&f=xml", url4);
        // Test case 5:  Null productLine
        String url5 = query.KeywordSearchGenerator("test5", null, "typeE", "5");
        assertEquals("https://example.com/search?t=12345&dev-t=DSB0XDDW1GQ3S&KeywordSearch=test5&mode=&type=typeE&page=5&f=xml", url5);
        // Test case 6: Null type
        String url6 = query.KeywordSearchGenerator("test6", "productF", null, "6");
        assertEquals("https://example.com/search?t=12345&dev-t=DSB0XDDW1GQ3S&KeywordSearch=test6&mode=productF&type=&page=6&f=xml", url6);
        // Test case 7: Null page
        String url7 = query.KeywordSearchGenerator("test7", "productG", "typeG", null);
        assertEquals("https://example.com/search?t=12345&dev-t=DSB0XDDW1GQ3S&KeywordSearch=test7&mode=productG&type=typeG&page=&f=xml", url7);
    }
}
