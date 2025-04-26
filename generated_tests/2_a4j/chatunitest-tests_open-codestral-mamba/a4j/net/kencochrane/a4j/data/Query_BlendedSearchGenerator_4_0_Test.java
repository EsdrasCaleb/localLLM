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

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Query query;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testBlendedSearchGenerator() {
        String type = "product";
        String searchTerm = "laptop";
        String expectedUrl = "https://server.com?t=associatesID&dev-t=DSB0XDDW1GQ3S&BlendedSearch=laptop&type=product&f=xml";
        when(jawsUtil.encodeString(searchTerm)).thenReturn("laptop");
        String actualUrl = query.BlendedSearchGenerator(type, searchTerm);
        assertEquals(expectedUrl, actualUrl);
    }
}
