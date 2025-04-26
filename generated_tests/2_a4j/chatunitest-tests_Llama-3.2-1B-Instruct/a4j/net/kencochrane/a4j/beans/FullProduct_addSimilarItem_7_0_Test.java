// Test method
package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.util.List;
import java.io.Serializable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class FullProduct_addSimilarItem_7_0_Test {

    @Mock
    private ProductDetails details;

    @InjectMocks
    private FullProduct fullProduct;

    @Test
    public void testAddSimilarItem() {
        // Arrange
        MiniProduct item = new MiniProduct();
        fullProduct.setAccessories(new ArrayList<>());
        // Act
        fullProduct.addSimilarItem(item);
        // Assert
        List<MiniProduct> expectedSimilarItems = new ArrayList<>();
        expectedSimilarItems.add(item);
        assertTrue(fullProduct.getSimilarItems().contains(item));
    }
}
