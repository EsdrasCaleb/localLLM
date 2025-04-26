package net.kencochrane.a4j.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
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
import java.util.Properties;

public class Query_queryGenerator_1_0_Test {

    // Test method for "queryGenerator(String, String, String, String, ArrayList)"
    @Test
    public void testQueryGenerator() {
        // Arrange
        String serverURL = "http://example.com";
        String associatesID = "12345";
        String token = "DSB0XDDW1GQ3S";
        String searchType = "name";
        String type = "product";
        String page = "1";
        String offer = "10";
        ArrayList searchValues = new ArrayList<>(Arrays.asList("apple", "banana"));
        String expectedURL = "http://example.com?t=12345&dev-t=DSB0XDDW1GQ3S&name=apple%2Cbanana&type=product&offerpage=1&offer=10&f=xml";
        // Act
        Query query = new Query();
        String actualURL = query.queryGenerator(searchType, type, page, offer, searchValues);
        // Assert
        assertEquals(expectedURL, actualURL);
    }
}
