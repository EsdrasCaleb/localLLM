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

class Query_ClearCart_10_1_Test {

    @Test
    void testClearCart() {
        Query query = new Query();
        String cartId = "12345";
        String hmac = "1234567890";
        String expectedURL = "http://example.com/clear?f=xml&dev-t=1234567890&t=1234567890&CartId=12345&Hmac=12345678901234567890";
        String actualURL = query.ClearCart(cartId, hmac);
        assertEquals(expectedURL, actualURL);
    }
}
