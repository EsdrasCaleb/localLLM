package net.kencochrane.a4j;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

@ExtendWith(MockitoExtension.class)
public class A4j_ActorSearch_3_2_Test {

    @Mock
    private Search search;

    @InjectMocks
    private A4j a4j;

    @Test
    public void testActorSearch() {
        String actorName = "John";
        String mode = "search";
        String page = "1";
        ProductInfo productInfo = new ProductInfo();
        when(search.ActorSearch(actorName, mode, page)).thenReturn(productInfo);
        ProductInfo result = a4j.ActorSearch(actorName, mode, page);
        assertEquals(productInfo, result);
    }
}
