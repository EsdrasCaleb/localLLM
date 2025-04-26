package net.kencochrane.a4j.data;

import java.util.ArrayList;
import java.util.Arrays;
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

@DisplayName("Query.AddToExistingCart(String, String, String, String)")
public class Query_AddToExistingCart_9_0_Test {

    Query query = new Query();

    @Test
    void testAddToExistingCart() {
        String result = query.AddToExistingCart("B00000000000", "1", "CARTID", "HMAC");
        assertEquals("http://xml.amazon.com/onca/xml3?ShoppingCart=add&f=xml&dev-t=DSB0XDDW1GQ3S&t=12345678901234567890123456789012&Asin.B00000000000=1&CartId=CARTID&Hmac=HMAC", result);
    }
}
