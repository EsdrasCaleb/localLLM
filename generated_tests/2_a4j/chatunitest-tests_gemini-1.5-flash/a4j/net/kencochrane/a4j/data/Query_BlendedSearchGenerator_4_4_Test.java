package net.kencochrane.a4j.data;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.util.LoadProperties;
import net.kencochrane.a4j.util.a4jUtil;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.Properties;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.LoadProperties;

@ExtendWith(MockitoExtension.class)
class Query_BlendedSearchGenerator_4_4_Test {

    @Mock
    a4jUtil mockUtil;

    @Test
    void testBlendedSearchGenerator() throws Exception {
        Query query = new Query();
        Field utilField = query.getClass().getDeclaredField("jawsUtil");
        utilField.setAccessible(true);
        utilField.set(query, mockUtil);
        // Test case 1: Normal case
        String searchTerm1 = "test search";
        String type1 = "testType";
        when(mockUtil.encodeString(searchTerm1)).thenReturn("encodedTestSearch");
        setField(query, "serverURL", "http://test.com");
        setField(query, "associatesID", "12345");
        String expectedURL1 = "http://test.com?t=12345&dev-t=DSB0XDDW1GQ3S&BlendedSearch=encodedTestSearch&type=testType&f=xml";
        String actualURL1 = query.BlendedSearchGenerator(type1, searchTerm1);
        assertEquals(expectedURL1, actualURL1);
        // Test case 2: Empty search term
        String searchTerm2 = "";
        String type2 = "anotherType";
        when(mockUtil.encodeString(searchTerm2)).thenReturn("");
        String expectedURL2 = "http://test.com?t=12345&dev-t=DSB0XDDW1GQ3S&BlendedSearch=&type=anotherType&f=xml";
        String actualURL2 = query.BlendedSearchGenerator(type2, searchTerm2);
        assertEquals(expectedURL2, actualURL2);
        // Test case 3: Null search term
        String searchTerm3 = null;
        String type3 = "yetAnotherType";
        when(mockUtil.encodeString(searchTerm3)).thenReturn(null);
        String expectedURL3 = "http://test.com?t=12345&dev-t=DSB0XDDW1GQ3S&BlendedSearch=&type=yetAnotherType&f=xml";
        String actualURL3 = query.BlendedSearchGenerator(type3, searchTerm3);
        assertEquals(expectedURL3, actualURL3);
        // Test case 4: Null type
        String searchTerm4 = "test4";
        String type4 = null;
        when(mockUtil.encodeString(searchTerm4)).thenReturn("encodedTest4");
        String expectedURL4 = "http://test.com?t=12345&dev-t=DSB0XDDW1GQ3S&BlendedSearch=encodedTest4&type=&f=xml";
        String actualURL4 = query.BlendedSearchGenerator(type4, searchTerm4);
        assertEquals(expectedURL4, actualURL4);
        // Test case 5: Empty type
        String searchTerm5 = "test5";
        String type5 = "";
        when(mockUtil.encodeString(searchTerm5)).thenReturn("encodedTest5");
        String expectedURL5 = "http://test.com?t=12345&dev-t=DSB0XDDW1GQ3S&BlendedSearch=encodedTest5&type=&f=xml";
        String actualURL5 = query.BlendedSearchGenerator(type5, searchTerm5);
        assertEquals(expectedURL5, actualURL5);
    }

    private void setField(Object object, String fieldName, Object value) throws Exception {
        Field field = object.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(object, value);
    }
}
