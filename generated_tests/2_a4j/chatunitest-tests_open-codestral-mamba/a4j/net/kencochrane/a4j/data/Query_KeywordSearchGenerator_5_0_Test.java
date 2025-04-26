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

    private Query query;

    @BeforeEach
    void setUp() {
        query = new Query();
    }

    @Test
    void testKeywordSearchGenerator() {
        String searchTerm = "testTerm";
        String productLine = "testProductLine";
        String type = "testType";
        String page = "testPage";
        String expectedUrl = query.serverURL + "?" + "t=" + query.associatesID + "&" + "dev-t=" + query.token + "&" + "KeywordSearch=" + query.jawsUtil.encodeString(searchTerm) + "&" + "mode=" + productLine + "&" + "type=" + type + "&" + "page=" + page + "&" + "f=xml";
        String actualUrl = query.KeywordSearchGenerator(searchTerm, productLine, type, page);
        assertEquals(expectedUrl, actualUrl);
    }
}
