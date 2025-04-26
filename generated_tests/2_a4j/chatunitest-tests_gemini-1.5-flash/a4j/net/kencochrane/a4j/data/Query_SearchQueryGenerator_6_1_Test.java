package net.kencochrane.a4j.data;

import java.lang.reflect.Field;
import org.mockito.junit.jupiter.MockitoExtension;
import org.junit.jupiter.api.extension.ExtendWith;
import net.kencochrane.a4j.util.LoadProperties;
import net.kencochrane.a4j.util.a4jUtil;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.Properties;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.LoadProperties;

@ExtendWith(MockitoExtension.class)
class Query_SearchQueryGenerator_6_1_Test {

    @Mock
    a4jUtil mockUtil;

    @Test
    void testSearchQueryGeneratorAllParams() {
        Query query = new Query();
        try {
            Field serverURLField = Query.class.getDeclaredField("serverURL");
            serverURLField.setAccessible(true);
            serverURLField.set(query, "http://example.com/search");
            Field associatesIDField = Query.class.getDeclaredField("associatesID");
            associatesIDField.setAccessible(true);
            associatesIDField.set(query, "12345");
            when(mockUtil.encodeString("test")).thenReturn("encodedTest");
            Field utilField = Query.class.getDeclaredField("jawsUtil");
            utilField.setAccessible(true);
            utilField.set(query, mockUtil);
            String result = query.SearchQueryGenerator("keyword", "test", "exact", "product", "1", "dailydeal");
            assertEquals("http://example.com/search?t=12345&dev-t=DSB0XDDW1GQ3S&keyword=encodedTest&mode=exact&type=product&page=1&offer=dailydeal&f=xml", result);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Exception during reflection: " + e.getMessage());
        }
    }

    @Test
    void testSearchQueryGeneratorEmptySearchTerm() {
        Query query = new Query();
        try {
            Field serverURLField = Query.class.getDeclaredField("serverURL");
            serverURLField.setAccessible(true);
            serverURLField.set(query, "http://example.com/search");
            Field associatesIDField = Query.class.getDeclaredField("associatesID");
            associatesIDField.setAccessible(true);
            associatesIDField.set(query, "12345");
            when(mockUtil.encodeString("")).thenReturn("");
            Field utilField = Query.class.getDeclaredField("jawsUtil");
            utilField.setAccessible(true);
            utilField.set(query, mockUtil);
            String result = query.SearchQueryGenerator("keyword", "", "exact", "product", "1", "dailydeal");
            assertEquals("http://example.com/search?t=12345&dev-t=DSB0XDDW1GQ3S&keyword=&mode=exact&type=product&page=1&offer=dailydeal&f=xml", result);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Exception during reflection: " + e.getMessage());
        }
    }
}
