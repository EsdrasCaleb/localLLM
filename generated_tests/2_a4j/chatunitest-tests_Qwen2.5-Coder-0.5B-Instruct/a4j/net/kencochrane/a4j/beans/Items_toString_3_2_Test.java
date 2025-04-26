package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.Serializable;
import java.util.ArrayList;

class Items_toString_3_2_Test {

    @Test
    public void testToString() {
        // Arrange
        Items items = mock(Items.class);
        when(items.getItemsArrayList()).thenReturn(new ArrayList<>());
        when(items.getItem()).thenReturn(null);
        // Act
        String result = items.toString();
        // Assert
        assertEquals("No Products", result);
    }
}
