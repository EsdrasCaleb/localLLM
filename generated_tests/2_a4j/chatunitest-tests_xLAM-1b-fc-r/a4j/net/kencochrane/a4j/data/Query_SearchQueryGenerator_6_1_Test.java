package net.kencochrane.a4j.data;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
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
public class Query_SearchQueryGenerator_6_1_Test {

    @InjectMocks
    Query query;

    @Test
    public void testSearchQueryGenerator() {
        String searchType = "products";
        String searchTerm = "test";
        String mode = "exact";
        String type = "products";
        String page = "1";
        String offer = "1";
        String expectedUrl = "http://example.com?t=associatesID&dev-t=token&type=products&page=1&offer=1&f=xml";
        String actualUrl = query.SearchQueryGenerator(searchType, searchTerm, mode, type, page, offer);
        assertEquals(expectedUrl, actualUrl);
    }
}
