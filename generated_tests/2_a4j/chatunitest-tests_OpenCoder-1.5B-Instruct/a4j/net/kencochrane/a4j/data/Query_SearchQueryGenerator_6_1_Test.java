package net.kencochrane.a4j.data;

import java.util.ArrayList;
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
import java.util.Properties;

public class Query_SearchQueryGenerator_6_1_Test {

    @Test
    public void testSearchQueryGenerator() {
        // Arrange
        String serverURL = "http://example.com";
        String associatesID = "12345";
        String searchType = "name";
        String searchTerm = "test";
        String mode = "exact";
        String type = "product";
        String page = "1";
        String offer = "free";
        a4jUtil jawsUtil = Mockito.mock(a4jUtil.class);
        ArrayList<String> searchValues = new ArrayList<>();
        searchValues.add(searchTerm);
        when(jawsUtil.encodeString(searchTerm)).thenReturn(searchTerm);
        Query query = new Query();
        query.serverURL = serverURL;
        query.associatesID = associatesID;
        query.jawsUtil = jawsUtil;
        query.searchType = searchType;
        query.type = type;
        query.page = page;
        query.offer = offer;
        query.searchValues = searchValues;
        // Act
        String result = query.SearchQueryGenerator(searchType, searchTerm, mode, type, page, offer);
        // Assert
        assertEquals("http://example.com?t=12345&dev-t=DSB0XDDW1GQ3S&name=test&mode=exact&type=product&page=1&offer=free&f=xml", result);
    }
}
