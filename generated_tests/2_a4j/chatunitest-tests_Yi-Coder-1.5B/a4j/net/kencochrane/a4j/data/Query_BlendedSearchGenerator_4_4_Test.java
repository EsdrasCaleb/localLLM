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

@ExtendWith(MockitoExtension.class)
public class Query_BlendedSearchGenerator_4_4_Test {

    // Test class
    @Test
    public void testBlendedSearchGenerator() {
        Query q = new Query();
        String url = q.BlendedSearchGenerator("1", "test");
        assertEquals("http://www.google.com/search?t=456999593&dev-t=DSB0XDDW1GQ3S&BlendedSearch=test%20%20&type=1&f=xml", url);
    }
}
