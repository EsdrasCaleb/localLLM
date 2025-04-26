package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class BlendedSearch_toString_3_2_Test {

    @Mock
    private BlendedSearch focal;

    @InjectMocks
    private BlendedSearch testObject;

    @Test
    public void testToString() {
        // Arrange
        List<ProductLine> productLines = new ArrayList<>();
        ProductLine productLine1 = new ProductLine();
        ProductLine productLine2 = new ProductLine();
        productLines.add(productLine1);
        productLines.add(productLine2);
        // Act
        String output = testObject.toString();
        // Assert
        assertEquals("## of productLines = 2\nproductLines = [[ProductLine@...], [ProductLine@...]]", output);
    }
}
