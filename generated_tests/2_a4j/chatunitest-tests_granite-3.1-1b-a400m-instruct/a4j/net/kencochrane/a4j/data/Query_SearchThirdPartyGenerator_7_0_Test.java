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
    public void testSearchThirdPartyGenerator() {
        // Arrange
        String sellerId = "12345";
        String type = "example";
        String page = "1";
        String status = "active";
        Query query = new Query();
        // Act
        String result = query.SearchThirdPartyGenerator(sellerId, type, page, status);
        // Assert
        assertEquals("http://example.com/?t=12345&dev-t=DSB0XDDW1GQ3S&SellerSearch=12345&type=example&page=1&offerstatus=active&f=xml", result);
    }
}
