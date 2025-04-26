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

public class Query_SearchQueryGenerator_6_0_Test {

    @Test
    void testSearchQueryGenerator() {
        Query query = new Query();
        String expected = "https://localhost:8080/search?t=1234567890&dev-t=DSB0XDDW1GQ3S&searchType=test&searchTerm=test&mode=search&type=test&page=1&offer=test&f=xml";
        String actual = query.SearchQueryGenerator("test", "test", "search", "test", "1", "test");
        assertEquals(expected, actual);
    }
}
