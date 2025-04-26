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

public class Query_BlendedSearchGenerator_4_0_Test {

    @Test
    void BlendedSearchGenerator() {
        Query query = new Query();
        String expectedUrl = "https://your-server-url/BlendedSearch?t=your-associates-id&dev-t=DSB0XDDW1GQ3S&BlendedSearch=search%20term&type=your-type&f=xml";
        String actualUrl = query.BlendedSearchGenerator("your-type", "search term");
        assertEquals(expectedUrl, actualUrl);
    }
}
