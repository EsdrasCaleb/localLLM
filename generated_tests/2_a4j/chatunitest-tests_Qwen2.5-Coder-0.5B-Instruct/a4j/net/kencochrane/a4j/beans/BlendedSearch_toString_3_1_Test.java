package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class BlendedSearch_toString_3_1_Test {

    @Test
    public void testToString() {
        // Arrange
        BlendedSearch search = mock(BlendedSearch.class);
        ArrayList productLines = mock(ArrayList.class);
        when(search.getProductLinesArrayList()).thenReturn(productLines);
        // Act
        String result = search.toString();
        // Assert
        assertEquals("# of productLines = 1\n" + "productLines: ProductLine [id=1, name=Apple, price=10]\n" + "productLines: ProductLine [id=2, name=Banana, price=20]\n", result);
    }
}
