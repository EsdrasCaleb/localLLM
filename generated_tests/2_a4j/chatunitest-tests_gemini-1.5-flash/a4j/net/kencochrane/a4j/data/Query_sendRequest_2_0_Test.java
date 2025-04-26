package net.kencochrane.a4j.data;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.net.URL;
import java.net.URLConnection;
import java.net.HttpURLConnection;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.LoadProperties;
import net.kencochrane.a4j.util.a4jUtil;
import java.util.Properties;

@ExtendWith(MockitoExtension.class)
public class Query_sendRequest_2_0_Test {

    @Test
    void testSendRequest_invalidUrl_throwsException() {
        Query query = new Query();
        assertThrows(IOException.class, () -> query.sendRequest("invalid-url"));
    }
}
