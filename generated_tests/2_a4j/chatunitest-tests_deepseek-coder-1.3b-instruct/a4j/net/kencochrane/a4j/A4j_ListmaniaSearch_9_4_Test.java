package net.kencochrane.a4j;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

public class A4j_ListmaniaSearch_9_4_Test {

    @Test
    public void testListmaniaSearch() {
        // Arrange
        String listId = "123";
        Search searchMock = mock(Search.class);
        when(searchMock.ListmaniaSearch(listId)).thenReturn(new ProductInfo());
        A4j a4j = new A4j();
        // Act
        ProductInfo result = a4j.ListmaniaSearch(listId);
        // Assert
        assertEquals(new ProductInfo(), result);
    }
}
