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

// Main class
public class Query_BlendedSearchGenerator_4_0_Test {

    // Test method
    @Test
    public void testBlendedSearchGenerator() {
        // Mocking the a4jUtil class
        a4jUtil mockA4jUtil = mock(a4jUtil.class);
        // Setting up the behavior of the mock
        when(mockA4jUtil.encodeString("Java Programming")).thenReturn("Java%20Programming");
        // Creating an instance of the Query class
        Query query = new Query();
        // Setting the a4jUtil field to the mock
        query.jawsUtil = mockA4jUtil;
        // Invoking the BlendedSearchGenerator method
        String result = query.BlendedSearchGenerator("book", "Java Programming");
        // Asserting that the result is as expected
        assertEquals("http://example.com?t=12345&dev-t=DSB0XDDW1GQ3S&BlendedSearch=Java%20Programming&type=book&f=xml", result);
    }
}
