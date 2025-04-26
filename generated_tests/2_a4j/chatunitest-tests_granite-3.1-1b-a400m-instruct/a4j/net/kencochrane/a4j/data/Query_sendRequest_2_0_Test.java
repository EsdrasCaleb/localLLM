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

public class Query_sendRequest_2_0_Test {

    @Test
    public void testSendRequest() throws Exception {
        // Arrange
        Query query = new Query();
        String string = "http://example.com";
        // Act
        String response = query.sendRequest(string);
        // Assert
        Assertions.assertEquals("Expected response from 'http://example.com'", response, "Response does not match expected string");
    }
}
