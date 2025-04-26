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
public class Query_SearchThirdPartyGenerator_7_0_Test {

    @InjectMocks
    Query query;

    @Test
    public void testSearchThirdPartyGenerator() {
        String sellerId = "123456";
        String type = "product";
        String page = "1";
        String status = "active";
        String expectedUrl = "http://example.com/search?t=A4J&dev-t=DSB0XDDW1GQ3S&SellerSearch=123456&type=product&page=1&offerstatus=active&f=xml";
        String actualUrl = query.SearchThirdPartyGenerator(sellerId, type, page, status);
        assertEquals(expectedUrl, actualUrl);
    }
}
