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

public class Query_KeywordSearchGenerator_5_1_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Query query;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testKeywordSearchGenerator() throws Exception {
        when(jawsUtil.encodeString("example")).thenReturn("encoded_example");
        String searchTerm = "example";
        String productLine = "product_line";
        String type = "type";
        String page = "page";
        String expectedUrl = "http://serverURL?t= associatesID&dev-t=DSB0XDDW1GQ3S&KeywordSearch=encoded_example&mode=product_line&type=type&page=page&f=xml";
        assertEquals(expectedUrl, query.KeywordSearchGenerator(searchTerm, productLine, type, page));
    }
}
