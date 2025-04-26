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
public class BlendedSearch_printProductList_4_2_Test {

    @InjectMocks
    private BlendedSearch blendedSearch;

    @Mock
    private ProductLine productLine;

    @Test
    void testPrintProductList() {
        // Arrange
        List<ProductLine> productLines = new ArrayList<>();
        productLines.add(productLine);
        productLines.add(productLine);
        productLines.add(productLine);
        // Act
        String output = blendedSearch.printProductList();
        // Assert
        assert output.contains("Product Line 1: Name: Product 1, Price: 10.99, Quantity: 5\n");
        assert output.contains("Product Line 2: Name: Product 2, Price: 12.99, Quantity: 3\n");
        assert output.contains("Product Line 3: Name: Product 3, Price: 15.99, Quantity: 2\n");
        assert output.contains("# of productLines = 3\n");
    }
}
