package net.kencochrane.a4j.beans;

import static org.junit.Assert.*;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

@RunWith(MockitoJUnitRunner.class)
public class RecentlyViewed_addProduct_0_3_Test {

    @Test(timeout = 1000)
    public void testAddProduct() throws Exception {
        RecentlyViewed underTest = Mockito.mock(RecentlyViewed.class);
        MiniProduct prod = Mockito.mock(MiniProduct.class);
        Mockito.when(prod.getAsin()).thenReturn("123456789");
        underTest.addProduct(prod);
        Mockito.verify(underTest).addProduct(prod);
    }
}
