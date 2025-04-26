package net.kencochrane.a4j.data;

import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
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
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.Properties;

public class Query_AddtoCart_8_0_Test {

    @Test
    public void testAddtoCart() throws Exception {
        // Arrange
        String ASIN = "B08GJRQGK";
        String quantity = "1";
        Query query = Mockito.mock(Query.class);
        URL expectedURL = new URL("https://www.amazon.com/gp/cart/add.html?ASIN=" + URLEncoder.encode(ASIN, StandardCharsets.UTF_8) + "&quantity=" + quantity);
        // Act
        String result = query.AddtoCart(ASIN, quantity);
        // Assert
        assertEquals(expectedURL.toString(), result);
    }
}
