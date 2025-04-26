package net.kencochrane.a4j.DAO;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { Search_Generic_2_4_Test.class, Search_UpcSearch_8_1_Test.class, Search_WishListSearch_10_2_Test.class, Search_SimilaritesSearch_12_0_Test.class })
public class Search_Suite {
}
