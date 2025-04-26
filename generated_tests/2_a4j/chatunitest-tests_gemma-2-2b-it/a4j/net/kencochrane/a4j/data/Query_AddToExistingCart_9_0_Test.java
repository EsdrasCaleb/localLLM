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

public class Query_AddToExistingCart_9_0_Test {

    @Test
    void AddToExistingCartTest() {
        Query query = new Query();
        String expected = "http://xml.amazon.com/onca/xml3?ShoppingCart=add&f=xml&dev-t=DSB0XDDW1GQ3S&t=1234567890&Asin.A1234567890=1&CartId=1234567890&Hmac=...";
        String actual = query.AddToExistingCart("A1234567890", "1", "1234567890", "...");
        assertEquals(expected, actual);
    }
}
