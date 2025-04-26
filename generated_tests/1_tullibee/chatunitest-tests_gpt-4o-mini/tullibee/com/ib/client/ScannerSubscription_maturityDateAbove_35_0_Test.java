package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateAbove_35_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMaturityDateAbove() {
        // Test with a standard date
        String date1 = "2023-12-31";
        scannerSubscription.maturityDateAbove(date1);
        assertEquals(date1, getPrivateField(scannerSubscription, "m_maturityDateAbove"));
        // Test with an empty string
        String date2 = "";
        scannerSubscription.maturityDateAbove(date2);
        assertEquals(date2, getPrivateField(scannerSubscription, "m_maturityDateAbove"));
        // Test with null
        String date3 = null;
        scannerSubscription.maturityDateAbove(date3);
        assertEquals(date3, getPrivateField(scannerSubscription, "m_maturityDateAbove"));
    }

    private Object getPrivateField(ScannerSubscription scannerSubscription, String fieldName) {
        try {
            java.lang.reflect.Field field = ScannerSubscription.class.getDeclaredField(fieldName);
            field.setAccessible(true);
            return field.get(scannerSubscription);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
