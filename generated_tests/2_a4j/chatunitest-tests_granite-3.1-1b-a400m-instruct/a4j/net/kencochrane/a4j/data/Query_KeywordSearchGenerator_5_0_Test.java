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
    void testKeywordSearchGenerator() {
        // Arrange
        String searchTerm = "example";
        String productLine = "electronics";
        String type = "electronics";
        String page = "1";
        // Setup
        Query query = new Query();
        // Act
        String result = query.KeywordSearchGenerator(searchTerm, productLine, type, page);
        // Assert
        assertEquals("http://example.com?t=example&dev-t=DSB0XDDW1GQ3S&KeywordSearch=example&mode=electronics&page=1&offer=f=xml", result);
    }
}
