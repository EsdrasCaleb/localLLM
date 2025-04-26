package com.ib.client;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

public class EWrapperMsgGenerator_tickPrice_0_1_Test {

    @Mock
    private TickType mockTickType;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        // Mocking static method using reflection
        try {
            Method getFieldMethod = TickType.class.getDeclaredMethod("getField", int.class);
            getFieldMethod.setAccessible(true);
            // Ensure the method is loaded
            getFieldMethod.invoke(null, 0);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testTickPriceWithAutoExecute() {
        int tickerId = 12345;
        int field = 1;
        double price = 123.45;
        int canAutoExecute = 1;
        // Mock the TickType.getField method
        try {
            Method getFieldMethod = TickType.class.getDeclaredMethod("getField", int.class);
            getFieldMethod.setAccessible(true);
            when((String) getFieldMethod.invoke(null, field)).thenReturn("BID");
        } catch (Exception e) {
            e.printStackTrace();
        }
        String result = EWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute);
        assertEquals("id=12345  BID=123.45 canAutoExecute", result);
    }

    @Test
    public void testTickPriceWithoutAutoExecute() {
        int tickerId = 67890;
        int field = 2;
        double price = 678.90;
        int canAutoExecute = 0;
        // Mock the TickType.getField method
        try {
            Method getFieldMethod = TickType.class.getDeclaredMethod("getField", int.class);
            getFieldMethod.setAccessible(true);
            when((String) getFieldMethod.invoke(null, field)).thenReturn("ASK");
        } catch (Exception e) {
            e.printStackTrace();
        }
        String result = EWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute);
        assertEquals("id=67890  ASK=678.9 noAutoExecute", result);
    }
}
