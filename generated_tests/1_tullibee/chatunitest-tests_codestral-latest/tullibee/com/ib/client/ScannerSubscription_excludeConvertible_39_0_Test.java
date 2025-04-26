package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ScannerSubscription_excludeConvertible_39_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testExcludeConvertible() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String expected = "excludeConvertibleCriteria";
        // Act
        scannerSubscription.excludeConvertible(expected);
        // Assert
        Field field = ScannerSubscription.class.getDeclaredField("m_excludeConvertible");
        field.setAccessible(true);
        String actual = (String) field.get(scannerSubscription);
        assertEquals(expected, actual);
    }

    @Test
    public void testExcludeConvertibleNull() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String expected = null;
        // Act
        scannerSubscription.excludeConvertible(expected);
        // Assert
        Field field = ScannerSubscription.class.getDeclaredField("m_excludeConvertible");
        field.setAccessible(true);
        String actual = (String) field.get(scannerSubscription);
        assertNull(actual);
    }
}
