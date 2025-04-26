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

public class Query_SearchQueryGenerator_6_3_Test {

    @Test
    public void testSearchQueryGenerator() {
        Query query = new Query();
        query.serverURL = "http://example.com";
        query.associatesID = "12345";
        query.token = "DSB0XDDW1GQ3S";
        query.searchType = "product";
        query.type = "xml";
        query.page = "1";
        query.offer = "12345";
        String expectedQuery = "http://example.com?t=12345&dev-t=DSB0XDDW1GQ3S&product=searchTerm&mode=exact&type=xml&page=1&offer=12345&f=xml";
        String result = query.SearchQueryGenerator("product", "searchTerm", "exact", "xml", "1", "12345");
        assertEquals(expectedQuery, result);
    }
}
