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

public class Query_AddtoCart_8_1_Test {

    private Query query;

    @BeforeEach
    public void setUp() {
        query = Mockito.spy(new Query());
    }

    @Test
    public void testAddtoCart() {
        String asin = "B07VGRJDFY";
        String quantity = "2";
        String expectedUrl = "https://server.com?ShoppingCart=add&f=xml&dev-t=DSB0XDDW1GQ3S&t=associateID&Asin." + asin + "=" + quantity;
        String actualUrl = query.AddtoCart(asin, quantity);
        assertEquals(expectedUrl, actualUrl);
    }
}
