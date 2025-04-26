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
public class RecentlyViewed_addProduct_0_0_Test {

    @Mock
    private MiniProduct mockMiniProduct;

    @InjectMocks
    private RecentlyViewed recentlyViewed;

    @Test
    public void testAddProduct() {
        // Arrange
        List<MiniProduct> products = new ArrayList<>();
        products.add(new MiniProduct());
        // Act
        recentlyViewed.addProduct(mockMiniProduct);
        // Assert
        verify(mockMiniProduct).getAsin();
    }
}
