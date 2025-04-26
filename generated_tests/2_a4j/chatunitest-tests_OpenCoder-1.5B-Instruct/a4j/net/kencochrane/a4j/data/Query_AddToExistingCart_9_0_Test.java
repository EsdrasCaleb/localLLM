package net.kencochrane.a4j.data;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
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

public class Query_AddToExistingCart_9_0_Test {

    @Test
    public void testAddToExistingCart() {
        Query query = new Query();
        String expectedURL = "http://example.com/onca/xml3?ShoppingCart=add&f=xml&dev-t=DSB0XDDW1GQ3S&t=12345&Asin.ASIN123=1&CartId=67890&Hmac=abcde";
        String actualURL = query.AddToExistingCart("ASIN123", "1", "67890", "abcde");
        assertEquals(expectedURL, actualURL);
    }
}
