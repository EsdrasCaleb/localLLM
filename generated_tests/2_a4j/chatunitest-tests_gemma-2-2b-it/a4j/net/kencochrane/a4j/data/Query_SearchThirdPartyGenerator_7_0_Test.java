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

public class Query_SearchThirdPartyGenerator_7_0_Test {

    @Test
    void testSearchThirdPartyGenerator() {
        Query query = mock(Query.class);
        when(query.SearchThirdPartyGenerator("sellerId", "type", "page", "status")).thenReturn("http://example.com/thirdparty/search");
        String result = query.SearchThirdPartyGenerator("sellerId", "type", "page", "status");
        assertEquals("http://example.com/thirdparty/search", result);
    }
}
