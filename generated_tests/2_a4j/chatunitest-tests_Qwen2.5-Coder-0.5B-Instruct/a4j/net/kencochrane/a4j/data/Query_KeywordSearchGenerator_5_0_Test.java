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

class Query_KeywordSearchGenerator_5_0_Test {

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void keywordSearchGeneratorTest() {
        String searchTerm = "test search";
        String productLine = "product123";
        String type = "productType";
        String page = "1";
        String expectedUrl = "https://example.com/search?t=associatesID&dev-t=token&KeywordSearch=test search&mode=product123&type=productType&page=1&f=xml";
        Query query = new Query();
        String result = query.KeywordSearchGenerator(searchTerm, productLine, type, page);
        assertEquals(expectedUrl, result);
    }
}
