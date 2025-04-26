package net.kencochrane.a4j.data;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;
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

@RunWith(MockitoJUnitRunner.class)
public class Query_AddtoCart_8_0_Test {

    @Mock
    private Query query;

    @Test
    public void testAddtoCart() throws Exception {
        assertEquals("<?xml version=\"1.0\" encoding=\"utf-8\"?><ShoppingCart xmlns=\"http://mws.amazonservices.com/schema/Cart/2013-09-01\"><AddItemRequest><Item><ASIN>B000000000</ASIN><Quantity>1</Quantity></Item></AddItemRequest></ShoppingCart>", query.AddtoCart("B000000000", "1"));
    }
}
