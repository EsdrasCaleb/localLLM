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

public class Query_KeywordSearchGenerator_5_0_Test {

    @Test
    void testKeywordSearchGenerator() {
        Query query = new Query();
        String expectedURL = "https://example.com/api/v1/search/keyword?t=your_associates_id&dev-t=DSB0XDDW1GQ3S&KeywordSearch=test_keyword&mode=your_product_line&type=your_type&page=your_page&f=xml";
        String actualURL = query.KeywordSearchGenerator("test_keyword", "your_product_line", "your_type", "your_page");
        assertEquals(expectedURL, actualURL);
    }
}
