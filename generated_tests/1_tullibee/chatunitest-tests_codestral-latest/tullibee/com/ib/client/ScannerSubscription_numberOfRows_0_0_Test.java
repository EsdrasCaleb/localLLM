package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_numberOfRows_0_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testNumberOfRows_DefaultValue() {
        assertEquals(ScannerSubscription.NO_ROW_NUMBER_SPECIFIED, scannerSubscription.numberOfRows());
    }

    @Test
    public void testNumberOfRows_SetValue() throws NoSuchFieldException, IllegalAccessException {
        int expectedRows = 10;
        scannerSubscription.numberOfRows(expectedRows);
        Field field = ScannerSubscription.class.getDeclaredField("m_numberOfRows");
        field.setAccessible(true);
        int actualRows = (int) field.get(scannerSubscription);
        assertEquals(expectedRows, actualRows);
    }
}
