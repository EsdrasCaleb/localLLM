package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class BlendedSearch_toString_3_2_Test {

    @Test
    public void testToString() {
        // Arrange
        BlendedSearch obj = new BlendedSearch();
        ArrayList productLines = new ArrayList<>();
        productLines.add(new ProductLine());
        productLines.add(new ProductLine());
        // Act
        String result = obj.toString();
        // Assert
        assertEquals("productLines is null \n", result);
    }
}
