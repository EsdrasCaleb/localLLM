package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class BlendedSearch_printProductList_4_0_Test {

    @Test
    void printProductList_withValidInput() {
        // Arrange
        ProductLine productLine1 = new ProductLine();
        productLine1.setProductName("Product 1");
        productLine1.setProductDescription("Description 1");
        ProductLine productLine2 = new ProductLine();
        productLine2.setProductName("Product 2");
        productLine2.setProductDescription("Description 2");
        ArrayList<ProductLine> productLines = new ArrayList<>();
        productLines.add(productLine1);
        productLines.add(productLine2);
        BlendedSearch blendedSearch = new BlendedSearch();
        blendedSearch.setProductLine(productLines.toArray(new ProductLine[0]));
        // Act
        String result = blendedSearch.printProductList();
        // Assert
        assertTrue(result.contains("Product 1"));
        assertTrue(result.contains("Description 1"));
        assertTrue(result.contains("Product 2"));
        assertTrue(result.contains("Description 2"));
        assertTrue(result.contains("# of productLines = 2"));
    }

    @Test
    void printProductList_withNullInput() {
        // Arrange
        BlendedSearch blendedSearch = new BlendedSearch();
        // Act
        String result = blendedSearch.printProductList();
        // Assert
        assertEquals("productLines is null \n", result);
    }

    @Test
    void printProductList_withEmptyInput() {
        // Arrange
        BlendedSearch blendedSearch = new BlendedSearch();
        ArrayList<ProductLine> productLines = new ArrayList<>();
        blendedSearch.setProductLine(productLines.toArray(new ProductLine[0]));
        // Act
        String result = blendedSearch.printProductList();
        // Assert
        assertTrue(result.contains("# of productLines = 0"));
    }
}

// Dummy ProductLine class
class ProductLine {

    private String productName;

    private String productDescription;

    public String getProductName() {
        return productName;
    }

    public void setProductName(String productName) {
        this.productName = productName;
    }

    public String getProductDescription() {
        return productDescription;
    }

    public void setProductDescription(String productDescription) {
        this.productDescription = productDescription;
    }

    public String printProductList() {
        return productName + " " + productDescription;
    }
}
