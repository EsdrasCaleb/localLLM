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

class A4j_ListmaniaSearch_9_0_Test {

    @Test
    void testListmaniaSearch() {
        // Arrange
        String listId = "exampleListId";
        // Create an instance of A4j
        A4j a4j = mock(A4j.class);
        // Create an instance of Search
        Search search = mock(Search.class);
        // Set up the mock behavior
        when(a4j.ListmaniaSearch(listId)).thenReturn(new ProductInfo());
        // Invoke the method to test
        ProductInfo result = a4j.ListmaniaSearch(listId);
        // Verify the result
        assertEquals(new ProductInfo(), result);
        // Close the mock objects
        verify(a4j).ListmaniaSearch(listId);
        verify(search).ListmaniaSearch(listId);
    }
}
