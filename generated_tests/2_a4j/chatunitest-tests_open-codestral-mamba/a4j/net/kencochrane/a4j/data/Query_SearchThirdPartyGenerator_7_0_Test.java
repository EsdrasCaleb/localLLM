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

class Query_SearchThirdPartyGenerator_7_0_Test {

    private Query query;

    @BeforeEach
    void setUp() {
        query = spy(new Query());
    }

    @Test
    void testSearchThirdPartyGenerator() {
        String sellerId = "12345";
        String type = "product";
        String page = "1";
        String status = "active";
        String expectedUrl = "http://example.com?t=DSB0XDDW1GQ3S&dev-t=DUMMY_TOKEN&SellerSearch=12345&type=product&page=1&offerstatus=active&f=xml";
        String actualUrl = query.SearchThirdPartyGenerator(sellerId, type, page, status);
        assertEquals(expectedUrl, actualUrl);
    }
}
