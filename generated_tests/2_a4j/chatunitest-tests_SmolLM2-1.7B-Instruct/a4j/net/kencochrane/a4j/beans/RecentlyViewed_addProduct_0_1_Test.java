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
class RecentlyViewed_addProduct_0_1_Test {

    @Mock
    private RecentlyViewed recentlyViewed;

    @InjectMocks
    private RecentlyViewed underTest;

    @Test
    void testAddProduct() {
        // Arrange
        MiniProduct miniProd = new MiniProduct();
        underTest.addProduct(miniProd);
        // Act
        // Assert
    }
}
