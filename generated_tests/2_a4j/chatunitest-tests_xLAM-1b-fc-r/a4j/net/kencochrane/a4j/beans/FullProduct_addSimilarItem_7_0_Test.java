package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class FullProduct_addSimilarItem_7_0_Test {

    @Test
    public void addSimilarItem_addsItemToList_whenCalled() {
        // Arrange
        FullProduct fullProduct = new FullProduct();
        MiniProduct item = new MiniProduct();
        // Act
        fullProduct.addSimilarItem(item);
        // Assert
        List<MiniProduct> expected = Arrays.asList(item);
        assertEquals(expected, fullProduct.getSimilarItems());
    }
}
