package net.kencochrane.a4j.data;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.net.URL;
import java.net.URLConnection;
import java.net.URLStreamHandler;
import java.net.URLStreamHandlerFactory;
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
import java.util.ArrayList;
import java.util.Properties;

class Query_sendRequest_2_2_Test {

    @BeforeAll
    static void setUp() {
        // Mock URLStreamHandler to avoid actual network calls
        URLStreamHandlerFactory mockFactory = protocol -> new URLStreamHandler() {

            @Override
            protected URLConnection openConnection(URL u) throws IOException {
                URLConnection mockConnection = mock(URLConnection.class);
                when(mockConnection.getInputStream()).thenReturn(new ByteArrayInputStream("Mock Response".getBytes()));
                return mockConnection;
            }
        };
        URL.setURLStreamHandlerFactory(mockFactory);
    }

    @Test
    void testSendRequest() throws Exception {
        Query query = new Query();
        String response = query.sendRequest("http://example.com");
        assertEquals("Mock Response", response);
    }

    @Test
    void testSendRequestWithInvalidURL() {
        Query query = new Query();
        assertThrows(Exception.class, () -> query.sendRequest("invalid-url"));
    }
}
